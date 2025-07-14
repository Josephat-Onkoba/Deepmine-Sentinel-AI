# Complete Model Lifecycle Documentation
========================================

## 🎯 Overview
This document provides a complete understanding of where the Enhanced Dual-Branch Stability Predictor model is saved after training and how it's retrieved for predictions.

## 📁 Model Storage Structure

### Primary Model Files
```
deepmine_sentinel_ai/
├── models/                                          # Main model directory
│   ├── enhanced_dual_branch_model.keras            # Main trained model
│   ├── enhanced_dual_branch_model_components.pkl   # Preprocessors (scalers, encoders)
│   ├── enhanced_dual_branch_model_metadata.json    # Training metadata
│   ├── best_enhanced_model.keras                   # Best checkpoint during training
│   └── training_log.csv                           # Training history log
```

## 🔄 Complete Model Lifecycle

### 1. Training Phase

#### A. Via Django Management Command
```bash
python manage.py train_model --epochs 100 --batch-size 32
```

**Process:**
1. **Data Preparation**: Uses static CSV files or Django model data
2. **Model Training**: Enhanced dual-branch neural network with callbacks
3. **Automatic Checkpointing**: Best model saved to `models/best_enhanced_model.keras`
4. **Explicit Saving**: Final model saved to `models/enhanced_dual_branch_model.keras`
5. **Components Saving**: Preprocessors saved to `*_components.pkl`
6. **Metadata Saving**: Training info saved to `*_metadata.json`

**Key Files Created:**
- `models/enhanced_dual_branch_model.keras` - Complete trained model
- `models/enhanced_dual_branch_model_components.pkl` - Scalers and encoders
- `models/enhanced_dual_branch_model_metadata.json` - Training metadata

#### B. Via ML Service
```python
from core.ml_service import MLPredictionService

service = MLPredictionService()
result = service.train_model_with_current_data()
```

**Process:**
1. **Dynamic Data Creation**: Extracts data from Django models into temporary CSVs
2. **Model Training**: Same enhanced dual-branch architecture
3. **Automatic Saving**: Model automatically saved after successful training
4. **Metadata Generation**: Training metadata saved alongside model

### 2. Loading Phase

#### A. Via ML Service (Production)
```python
service = MLPredictionService()
# Automatically checks for and loads saved model
predictions = service.predict_comprehensive_stability(stope_id)
```

**Loading Priority:**
1. **Check for saved model**: `models/enhanced_dual_branch_model.keras`
2. **Load if exists**: Use `predictor.load_enhanced_model()`
3. **Fallback to training**: If no saved model, prepare for new training

#### B. Direct Model Loading
```python
from core.ml.models.dual_branch_stability_predictor import EnhancedDualBranchStabilityPredictor

model = EnhancedDualBranchStabilityPredictor(static_path, timeseries_path)
model.load_enhanced_model('models/enhanced_dual_branch_model.keras')
```

## 💾 Saving Mechanisms

### 1. Training Callbacks (During Training)
```python
tf.keras.callbacks.ModelCheckpoint(
    filepath='models/best_enhanced_model.keras',
    monitor='val_loss',
    save_best_only=True
)
```
- **Purpose**: Save best model during training
- **Location**: `models/best_enhanced_model.keras`
- **Trigger**: When validation loss improves

### 2. Explicit Model Saving (After Training)
```python
model.save_enhanced_model('models/enhanced_dual_branch_model.keras')
```
- **Purpose**: Save final trained model with all components
- **Location**: `models/enhanced_dual_branch_model.keras`
- **Components**: Model + preprocessors + metadata

### 3. Component Saving
```python
joblib.dump({
    'static_scaler': self.static_scaler,
    'timeseries_scaler': self.timeseries_scaler,
    'risk_label_encoder': self.risk_label_encoder,
    # ... other components
}, 'models/enhanced_dual_branch_model_components.pkl')
```

## 🔍 Model Retrieval Process

### 1. ML Service Initialization
```python
def _load_model(self):
    # 1. Check if pre-trained model exists
    if os.path.exists(self.model_path):
        # Load saved model
        self.predictor.load_enhanced_model(self.model_path)
        return True
    
    # 2. If no saved model, prepare for training
    else:
        # Initialize for new training
        return self._prepare_for_training()
```

### 2. Model Health Check
```python
def is_model_trained(self):
    # 1. Check if model already loaded in memory
    if self._model_loaded:
        return True
    
    # 2. Check if saved model exists on disk
    if os.path.exists(self.model_path):
        return True
    
    # 3. Check if data available for training
    return self._can_create_enhanced_model_data()
```

## 📋 File Locations Summary

### Standard Paths (as implemented)
```python
BASE_DIR = '/home/jose/Desktop/Deepmine-Sentinel-AI/deepmine_sentinel_ai'

# Model files
MODEL_PATH = f'{BASE_DIR}/models/enhanced_dual_branch_model.keras'
COMPONENTS_PATH = f'{BASE_DIR}/models/enhanced_dual_branch_model_components.pkl'
METADATA_PATH = f'{BASE_DIR}/models/enhanced_dual_branch_model_metadata.json'

# Training checkpoints
CHECKPOINT_PATH = f'{BASE_DIR}/models/best_enhanced_model.keras'
TRAINING_LOG = f'{BASE_DIR}/models/training_log.csv'
```

### Directory Structure After Training
```
/home/jose/Desktop/Deepmine-Sentinel-AI/deepmine_sentinel_ai/
└── models/
    ├── enhanced_dual_branch_model.keras          # 🎯 Main model (PRODUCTION)
    ├── enhanced_dual_branch_model_components.pkl # 🔧 Preprocessors
    ├── enhanced_dual_branch_model_metadata.json  # 📊 Training info
    ├── best_enhanced_model.keras                 # 💎 Best checkpoint
    └── training_log.csv                          # 📈 Training history
```

## 🚀 Production Usage Flow

### 1. Training (One-time or Periodic)
```bash
# Train new model
python manage.py train_model --epochs 100

# Model automatically saved to models/enhanced_dual_branch_model.keras
```

### 2. Production Predictions
```python
# Django views automatically use saved model
from core.ml_service import MLPredictionService

service = MLPredictionService()
# ✅ Automatically loads from models/enhanced_dual_branch_model.keras
result = service.predict_comprehensive_stability(stope_id)
```

### 3. Model Updates
```python
# Re-train and automatically replace saved model
service = MLPredictionService()
training_result = service.train_model_with_current_data()
# ✅ New model automatically saved, replacing previous version
```

## ⚠️ Important Notes

### Model Persistence
- **Persistent**: Model survives server restarts, application redeployments
- **Location**: File system under `models/` directory
- **Components**: Model architecture + weights + preprocessors + metadata

### Loading Strategy
- **Lazy Loading**: Model loaded when first needed
- **Memory Caching**: Once loaded, stays in memory for performance
- **Fallback**: If no saved model, can train new one automatically

### Model Versioning
- **Current**: Single model file, replaced on retrain
- **Future**: Could implement versioning with timestamps
- **Metadata**: Training date and parameters stored in metadata file

## 🎉 Summary

**Where is the model saved?**
- Primary: `/home/jose/Desktop/Deepmine-Sentinel-AI/deepmine_sentinel_ai/models/enhanced_dual_branch_model.keras`
- Components: `/home/jose/Desktop/Deepmine-Sentinel-AI/deepmine_sentinel_ai/models/enhanced_dual_branch_model_components.pkl`

**How is it retrieved?**
- Automatically by `MLPredictionService` when making predictions
- Manual loading via `model.load_enhanced_model(filepath)`

**When is it saved?**
- After successful training via Django management command
- After successful training via ML service
- During training as checkpoints

**Production Ready?**
- ✅ Yes, model persists across application restarts
- ✅ Automatic loading in production predictions
- ✅ Complete model lifecycle implemented
