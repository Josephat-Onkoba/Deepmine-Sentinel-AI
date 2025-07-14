# DeepMine Sentinel AI: Enhanced Dual-Branch Neural Network for Underground Mining Stope Stability Prediction

## 1. Task Definition, Evaluation Protocol, and Data

### 1.1 Project Description

The DeepMine Sentinel AI project addresses a critical challenge in underground mining operations: **real-time prediction and monitoring of stope stability to prevent catastrophic failures and ensure worker safety**. Mining stopes (excavated underground chambers) are subject to complex geological stresses, rock degradation, and temporal changes that can lead to sudden collapses, resulting in equipment loss, production delays, and potential loss of life.

This project implements an advanced machine learning system that combines static geological features with temporal sensor data to provide:

1. **Current Stability Assessment**: Binary classification of stope stability (stable/unstable)
2. **Multi-horizon Risk Forecasting**: Prediction of future risk levels at 1, 3, 7, 14, and 30-day horizons
3. **Risk Level Classification**: Six-class categorization (stable, slight_elevated, elevated, high, critical, unstable)
4. **Uncertainty Quantification**: Confidence intervals for all predictions

The system processes heterogeneous data sources including geological surveys, structural parameters, and continuous sensor monitoring to provide comprehensive risk assessment capabilities for mining operations.

### 1.2 Task Definition and Background

Underground mining stope stability prediction is a well-established challenge in geotechnical engineering. The task involves analyzing multiple factors that influence rock mass behavior:

- **Geological Parameters**: Rock Quality Designation (RQD), rock type, structural orientation
- **Geometric Factors**: Stope dimensions, depth, hydraulic radius, dip angle
- **Support Systems**: Type, density, and installation status of ground support
- **Temporal Monitoring**: Vibration velocity, deformation rates, stress measurements, environmental conditions

Traditional approaches rely on empirical methods such as the Modified Stability Graph Method (Potvin, 1988) and numerical modeling techniques. However, these methods often fail to capture the complex, non-linear relationships between multiple variables and their temporal evolution.

**Reference**: Potvin, Y. (1988). "Empirical open stope design in Canada." PhD thesis, University of British Columbia. This seminal work established the foundation for quantitative stope stability assessment using stability graphs based on hydraulic radius and stability number calculations.

### 1.3 Dataset Description

The project utilizes two aligned datasets:

#### 1.3.1 Static Features Dataset
- **Size**: 21 stopes with 12 features each
- **Features**: 
  - Geological: RQD (25.96-88.51%), rock type (11 categories), depth (285.9-725.8m)
  - Structural: Dip angle (46.03-76.20°), direction (8 cardinal/intercardinal directions)
  - Geometric: Hydraulic radius (4.13-14.56m), undercut width (3.70-7.18m)
  - Support: Type (7 categories), density (0.32-0.86), installation status (binary)
- **Target**: Binary stability classification (stable=1, unstable=0)

#### 1.3.2 Temporal Features Dataset
- **Size**: 1,801 time-series records across all stopes
- **Temporal Resolution**: Daily measurements over approximately 85 days per stope
- **Features**:
  - Vibration velocity (0.92-1.23 mm/s)
  - Deformation rate (0.12-0.37 mm/day)
  - Stress measurements (19.6-23.5 MPa)
  - Temperature (14.9-17.1°C)
  - Humidity (63.5-69.8%)

### 1.4 Evaluation Protocol

The evaluation protocol implements a comprehensive assessment strategy:

#### 1.4.1 Data Splitting
- **Training/Validation Split**: 80/20 stratified split maintaining class balance
- **Cross-validation**: Stratified approach to handle class imbalance
- **Temporal Consistency**: Time-series sequences maintained during splitting

#### 1.4.2 Evaluation Metrics
- **Current Stability**: Binary cross-entropy loss, accuracy, precision, recall
- **Future Risk Prediction**: Sparse categorical cross-entropy for multi-class classification
- **Model Performance**: Validation loss monitoring with early stopping
- **Uncertainty Quantification**: Confidence interval assessment

#### 1.4.3 Evaluation Methodology
1. **Training Phase**: Model trained with multi-task learning approach
2. **Validation Phase**: Performance assessed on held-out validation set
3. **Model Selection**: Best weights restored based on validation loss
4. **Performance Reporting**: Comprehensive metrics across all prediction tasks

![Figure 1: Task Overview and Data Flow](plots/model_architecture_diagram.png)
*Figure 1: Overview of the dual-branch architecture processing static geological features and temporal sensor data for multi-horizon stability prediction*

---

## 2. Neural Network / Machine Learning Model

### 2.1 Model Architecture Overview

The Enhanced Dual-Branch Stability Predictor implements a sophisticated neural network architecture designed to handle the heterogeneous nature of mining stability data. The model combines domain knowledge from geotechnical engineering with advanced deep learning techniques to address the multi-faceted nature of stope stability prediction.

### 2.2 Dual-Branch Architecture

#### 2.2.1 Static Features Branch (Dense Feedforward Network)

The static branch processes time-invariant geological and structural parameters through a deep feedforward network:

```
Input Layer (12 features) 
    ↓
Dense Layer (128 units) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Dense Layer (128 units) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Dense Layer (64 units) + ReLU + BatchNorm
    ↓
Static Feature Representation (64-dimensional)
```

**Key Components**:
- **Input Processing**: Standardized geological parameters (RQD, rock type, depth, etc.)
- **Feature Engineering**: Categorical encoding for rock types, support systems, and directions
- **Regularization**: Batch normalization and dropout to prevent overfitting
- **Activation**: ReLU activation for non-linear feature learning

#### 2.2.2 Temporal Features Branch (LSTM with Attention)

The temporal branch processes time-series sensor data using advanced recurrent architecture:

```
Input Layer (sequence_length × 5 features)
    ↓
LSTM Layer 1 (128 units, return_sequences=True) + Dropout(0.3)
    ↓
LSTM Layer 2 (128 units, return_sequences=True) + Dropout(0.3)
    ↓
Residual Connection (if dimensions match)
    ↓
Multi-Head Attention (8 heads, key_dim=16)
    ↓
Layer Normalization
    ↓
Final LSTM (64 units, return_sequences=False) + BatchNorm + Dropout(0.3)
    ↓
Temporal Feature Representation (64-dimensional)
```

**Advanced Features**:
- **Multi-layer LSTM**: Captures complex temporal dependencies in sensor data
- **Residual Connections**: Enables deeper network training and gradient flow
- **Multi-Head Attention**: Focuses on critical temporal patterns and anomalies
- **Layer Normalization**: Stabilizes attention mechanism training

#### 2.2.3 Feature Fusion and Multi-Task Heads

The architecture combines both branches for comprehensive prediction:

```
Static Features (64-dim) + Temporal Features (64-dim)
    ↓
Concatenation Layer (128-dimensional)
    ↓
Shared Dense Layer (128 units) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Shared Dense Layer (64 units) + ReLU + BatchNorm
    ↓
Task-Specific Prediction Heads
```

**Multi-Task Output Structure**:
1. **Current Stability Head**: 
   - Dense(32) → Dropout(0.3) → Dense(1, sigmoid)
   - Output: Binary probability (stable/unstable)

2. **Future Risk Heads** (5 horizons: 1, 3, 7, 14, 30 days):
   - Each horizon: Dense(32) → Dropout(0.3) → Dense(6, softmax)
   - Output: Risk level probabilities (6 classes)

### 2.3 Loss Function and Training Methodology

#### 2.3.1 Multi-Task Loss Formulation

The model employs a weighted multi-task loss function:

```
L_total = λ₁ · L_current + Σᵢ λᵢ₊₁ · L_future_i

Where:
- L_current = Binary Cross-Entropy(y_current, ŷ_current)
- L_future_i = Sparse Categorical Cross-Entropy(y_future_i, ŷ_future_i)
- λ₁ = 1.0 (current stability weight)
- λᵢ₊₁ = 0.5 (future prediction weights)
```

**Loss Function Details**:
- **Current Stability**: Binary cross-entropy optimized for immediate safety assessment
- **Future Risk Prediction**: Sparse categorical cross-entropy for multi-class risk levels
- **Loss Weighting**: Higher weight on current predictions for immediate safety prioritization

#### 2.3.2 Training Configuration

- **Optimizer**: Adam with learning rate 0.001
- **Batch Size**: 16-32 samples (adaptive based on dataset size)
- **Epochs**: Up to 100 with early stopping (patience=15)
- **Learning Rate Scheduling**: ReduceLROnPlateau (factor=0.5, patience=8)
- **Regularization**: Dropout (0.3), Batch Normalization, L2 regularization

#### 2.3.3 Model Complexity and Parameters

- **Total Parameters**: Approximately 50,000-75,000 trainable parameters
- **Static Branch**: ~25,000 parameters
- **Temporal Branch**: ~30,000 parameters  
- **Fusion and Heads**: ~15,000 parameters

### 2.4 Key Innovations and Theoretical Foundation

#### 2.4.1 Attention Mechanism for Temporal Mining Data

The multi-head attention mechanism addresses a critical challenge in mining monitoring: **identifying critical temporal patterns that precede stability failures**. Unlike traditional LSTM approaches, attention allows the model to focus on specific time periods and sensor combinations that are most predictive of future instability.

**Mathematical Formulation**:
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V

Where Q, K, V are query, key, and value matrices derived from LSTM outputs
```

#### 2.4.2 Multi-Horizon Prediction Strategy

The model implements a novel approach to temporal risk assessment by predicting stability at multiple future horizons simultaneously. This design is inspired by weather forecasting models and addresses the practical need for both immediate (1-day) and long-term (30-day) planning in mining operations.

**Theoretical Justification**: Rock mass degradation follows complex temporal patterns with both sudden failures and gradual deterioration. Multi-horizon prediction captures both rapid changes and slow-developing instabilities.

### 2.5 Model References and Theoretical Foundation

1. **Vaswani, A., et al. (2017)**. "Attention is All You Need." Neural Information Processing Systems. *The foundational work on attention mechanisms that inspired our temporal pattern recognition approach.*

2. **Graves, A., & Schmidhuber, J. (2005)**. "Framewise phoneme classification with bidirectional LSTM and other neural network architectures." Neural Networks. *Established the effectiveness of LSTM networks for sequential pattern recognition, directly applicable to temporal mining sensor data.*

3. **He, K., et al. (2016)**. "Deep Residual Learning for Image Recognition." Computer Vision and Pattern Recognition. *Residual connection methodology adapted for our multi-layer LSTM architecture to enable deeper temporal feature learning.*

The model architecture represents a significant advancement over traditional empirical methods by combining domain-specific geological knowledge with state-of-the-art deep learning techniques, specifically designed for the unique challenges of underground mining stability prediction.

![Figure 2: Enhanced Dual-Branch Architecture](plots/enhanced_model_architecture.png)
*Figure 2: Detailed architecture showing the dual-branch design with static geological features processing, temporal sensor data analysis through LSTM and attention mechanisms, and multi-task prediction heads for current and future stability assessment*

---

## 3. Experiment

### 3.1 Research Questions

This study investigates three fundamental questions regarding the effectiveness of hybrid neural architectures for underground mining stope stability prediction:

**RQ1**: Can a time-aware LSTM model, conditioned on static stope characteristics, accurately predict stope risk levels over time?

**RQ2**: Does the inclusion of static geotechnical and design features improve risk prediction compared to using time-series sensor data alone?

**RQ3**: Can the model correctly differentiate between different risk levels (Stable, Elevated, High Risk, Unstable) and offer early warnings for potential instability?

These research questions address a critical gap in current mining safety systems, where traditional approaches either rely solely on static geological assessments or temporal monitoring in isolation, failing to capture the complex interactions between geological conditions and temporal degradation patterns.

### 3.2 Experimental Design

#### 3.2.1 Hypotheses

The experimental design is structured around three testable hypotheses that directly address the research questions:

**H1 - Hybrid Model Superiority**: The hybrid model (LSTM + static features) will outperform a standalone LSTM in classifying stope risk levels, demonstrating statistical significance (p < 0.05) in overall accuracy and F1-score metrics.

*Mathematical Basis*: The hybrid model leverages both the universal approximation theorem for feedforward networks (processing static features) and the representational power of recurrent networks for temporal sequences. The fusion of these complementary representations should theoretically capture more comprehensive patterns than either modality alone.

**H2 - Early Detection Capability**: The model will achieve high recall (>85%) for the "Unstable" class, indicating reliable early detection capability for critical safety scenarios.

*Algorithmic Justification*: The multi-horizon prediction architecture with attention mechanisms should enable the model to identify temporal patterns that precede instability events, providing early warning capabilities essential for mining safety.

**H3 - Risk Level Discrimination**: Incorporating both static and dynamic features will reduce confusion between "Elevated Risk" and "High Risk" levels, measured by a 20% improvement in class-specific precision compared to temporal-only models.

*Data/Storage Properties*: Static geological features provide contextual boundaries for risk assessment, while temporal features capture degradation dynamics. Their combination should improve the model's ability to distinguish between adjacent risk levels in the classification hierarchy.

#### 3.2.2 Independent Variables (Experimental Conditions)

The experiment manipulates three primary independent variables to test the hypotheses:

| Variable | Condition 1 | Condition 2 | Condition 3 |
|----------|-------------|-------------|-------------|
| **Model Architecture** | Temporal-Only LSTM | Static-Only Dense Network | Hybrid Dual-Branch |
| **Feature Input** | Time-series only (5 features) | Static features only (12 features) | Combined (17 features) |
| **Prediction Scope** | Current stability only | Current stability only | Multi-horizon (1,3,7,14,30 days) |

**Detailed Experimental Conditions**:

1. **Condition 1 - Temporal-Only Baseline**: 
   - Architecture: 2-layer LSTM (128 units each) + Dense output
   - Input: Vibration velocity, deformation rate, stress, temperature, humidity
   - Output: Binary stability classification

2. **Condition 2 - Static-Only Baseline**:
   - Architecture: 3-layer Dense network (128, 128, 64 units)
   - Input: RQD, rock type, depth, dip, direction, hydraulic radius, support characteristics
   - Output: Binary stability classification

3. **Condition 3 - Hybrid Enhanced Model**:
   - Architecture: Dual-branch with attention mechanism
   - Input: Combined static and temporal features
   - Output: Current stability + multi-horizon risk predictions

#### 3.2.3 Control Variables (Biases and Modeling Assumptions)

To ensure experimental validity, the following variables are held constant across all conditions:

**Data Processing Controls**:
- Standardization: All features normalized using StandardScaler with fit on training data
- Sequence Length: 30-day temporal windows for all time-series inputs
- Train/Validation Split: 80/20 stratified split maintaining class balance
- Random Seed: Fixed at 42 for reproducible results

**Training Controls**:
- Optimizer: Adam with learning rate 0.001
- Batch Size: 16 samples
- Early Stopping: Patience of 15 epochs based on validation loss
- Loss Function: Binary cross-entropy for stability prediction
- Regularization: Dropout (0.3) and batch normalization applied consistently

**Hardware/Software Controls**:
- TensorFlow 2.x framework
- Python 3.8+ environment
- Fixed computational resources for fair comparison

**Modeling Assumptions**:
- Temporal independence: No information leakage between validation sequences
- Feature relevance: All selected features contribute meaningful information
- Class balance: Stratified sampling maintains representative class distributions

#### 3.2.4 Dependent Variables (Results Analysis)

The experiment measures multiple dependent variables to comprehensively evaluate model performance:

**Primary Performance Metrics**:
- **Overall Accuracy**: Percentage of correct predictions across all classes
- **Class-Specific Metrics**: Precision, Recall, and F1-score for each risk level
- **Confusion Matrix Analysis**: Detailed error patterns between risk categories
- **Area Under ROC Curve (AUC)**: Model's discriminative ability

**Temporal Analysis Metrics**:
- **Multi-Horizon Accuracy**: Performance across 1, 3, 7, 14, 30-day predictions
- **Early Warning Capability**: Time-to-detection for unstable classifications
- **Prediction Consistency**: Temporal coherence in risk level transitions

**Learning Dynamics**:
- **Training/Validation Loss Curves**: Convergence behavior and overfitting assessment
- **Epochs to Convergence**: Training efficiency comparison
- **Feature Importance Analysis**: Attention weights and gradient-based importance

### 3.3 Methodology

#### 3.3.1 Implementation Framework

**Code Base**: The experiment utilizes a custom implementation built on TensorFlow 2.x, specifically designed for mining stability prediction:

```python
# Core implementation structure
deepmine_sentinel_ai/
├── core/ml/models/
│   ├── dual_branch_stability_predictor.py  # Main hybrid model
│   ├── temporal_only_model.py              # LSTM baseline
│   └── static_only_model.py                # Dense baseline
├── experiments/
│   ├── run_baseline_comparison.py          # Experimental controller
│   └── evaluation_metrics.py               # Comprehensive metrics
└── data/
    ├── stope_static_features_aligned.csv   # Static geological data
    └── stope_timeseries_data_aligned.csv   # Temporal sensor data
```

#### 3.3.2 Data Processing Pipeline

**Preprocessing Requirements**:
1. **Static Features**: Categorical encoding for rock types, support systems, directions
2. **Temporal Features**: Sequence windowing with 30-day lookback periods
3. **Target Engineering**: Multi-class risk level encoding and horizon-specific labels
4. **Validation**: Temporal split ensuring no future information leakage

**Feature Engineering Modifications**:
- **Temporal Baseline**: Excludes static features, uses only sensor time-series
- **Static Baseline**: Excludes temporal sequences, uses geological snapshots
- **Hybrid Model**: Combines both modalities with specialized fusion architecture

#### 3.3.3 Experimental Protocol

**Training Procedure**:
1. **Data Preparation**: Load and preprocess datasets according to condition specifications
2. **Model Initialization**: Create architecture variant based on experimental condition
3. **Training Execution**: Fit model with consistent hyperparameters and monitoring
4. **Validation Assessment**: Evaluate on held-out validation set
5. **Statistical Analysis**: Compare metrics across conditions with significance testing

**Baseline Justification**:
The experimental design employs two complementary baselines:

- **Temporal-Only LSTM**: Represents current state-of-the-art in time-series mining monitoring
- **Static-Only Dense**: Represents traditional geological assessment approaches

This dual-baseline approach enables isolation of the contribution from each modality and validates the necessity of the hybrid architecture.

**Experimental Conditions Summary**:
The three-condition design systematically tests each component:
1. **Temporal capability** (Condition 1)
2. **Static feature utility** (Condition 2) 
3. **Hybrid integration benefit** (Condition 3)

This methodology enables direct comparison of individual model components and their combined effectiveness, providing clear answers to the stated research questions while maintaining scientific rigor through controlled experimentation.

### 3.4 Expected Outcomes and Analysis Plan

**Success Criteria**:
- H1 validation: Hybrid model achieves >10% improvement in F1-score over both baselines
- H2 validation: Unstable class recall exceeds 85% with hybrid model
- H3 validation: 20% reduction in Elevated/High Risk confusion compared to temporal-only baseline

**Statistical Analysis**: Paired t-tests for metric comparisons, McNemar's test for classification agreement, and effect size calculations using Cohen's d for practical significance assessment.

---
