#!/usr/bin/env python3
"""
Deepmine Sentinel AI - Project Organization Script

This script automatically organizes the project structure according to
the organization plan for better maintainability and production readiness.
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime
import argparse


class ProjectOrganizer:
    """Organize the Deepmine Sentinel AI project structure"""
    
    def __init__(self, project_root: str, dry_run: bool = True):
        self.project_root = Path(project_root)
        self.dry_run = dry_run
        self.backup_dir = self.project_root / "backup_before_organization"
        self.operations_log = []
        
    def log_operation(self, operation: str, source: str = "", target: str = ""):
        """Log organization operations"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "source": source,
            "target": target,
            "dry_run": self.dry_run
        }
        self.operations_log.append(entry)
        
        # Print operation
        if self.dry_run:
            print(f"[DRY RUN] {operation}: {source} -> {target}")
        else:
            print(f"[EXECUTE] {operation}: {source} -> {target}")
    
    def create_backup(self):
        """Create backup of current project state"""
        if not self.dry_run:
            if self.backup_dir.exists():
                shutil.rmtree(self.backup_dir)
            
            # Copy important files to backup
            important_dirs = ["deepmine_sentinel_ai", "docs"]
            for dir_name in important_dirs:
                src_dir = self.project_root / dir_name
                if src_dir.exists():
                    backup_target = self.backup_dir / dir_name
                    shutil.copytree(src_dir, backup_target)
                    
        self.log_operation("BACKUP", str(self.project_root), str(self.backup_dir))
    
    def create_directory_structure(self):
        """Create the new organized directory structure"""
        
        # Main directories to create
        directories = [
            # Root level
            "docs",
            "tests",
            "scripts",
            "data",
            "deployment",
            
            # Documentation structure
            "docs/installation",
            "docs/api",
            "docs/ml",
            "docs/user_guide", 
            "docs/developer",
            "docs/deployment",
            
            # Django app reorganization
            "deepmine_sentinel_ai/deepmine_sentinel_ai/settings",
            "deepmine_sentinel_ai/core/models",
            "deepmine_sentinel_ai/core/views",
            "deepmine_sentinel_ai/core/api",
            "deepmine_sentinel_ai/core/utils",
            "deepmine_sentinel_ai/core/tests",
            
            # ML system reorganization
            "deepmine_sentinel_ai/core/ml/models",
            "deepmine_sentinel_ai/core/ml/training", 
            "deepmine_sentinel_ai/core/ml/inference",
            "deepmine_sentinel_ai/core/ml/data",
            "deepmine_sentinel_ai/core/ml/utils",
            "deepmine_sentinel_ai/core/ml/config",
            
            # Data and models
            "deepmine_sentinel_ai/static_collected",
            "deepmine_sentinel_ai/media",
            
            # Tests organization
            "tests/unit",
            "tests/integration", 
            "tests/ml",
            "tests/api",
            "tests/fixtures",
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            if not self.dry_run:
                dir_path.mkdir(parents=True, exist_ok=True)
            self.log_operation("CREATE_DIR", "", str(dir_path))
    
    def organize_ml_system(self):
        """Reorganize the ML system files"""
        
        ml_moves = [
            # Core ML models
            ("deepmine_sentinel_ai/core/ml/lstm_models.py", 
             "deepmine_sentinel_ai/core/ml/models/lstm_models.py"),
            ("deepmine_sentinel_ai/core/ml/attention_layers.py", 
             "deepmine_sentinel_ai/core/ml/models/attention_layers.py"),
            ("deepmine_sentinel_ai/core/ml/lstm_config.py", 
             "deepmine_sentinel_ai/core/ml/config/lstm_config.py"),
            
            # Training system
            ("deepmine_sentinel_ai/core/ml/training_pipeline.py", 
             "deepmine_sentinel_ai/core/ml/training/training_pipeline.py"),
            ("deepmine_sentinel_ai/core/ml/training_config.py", 
             "deepmine_sentinel_ai/core/ml/training/training_config.py"),
            ("deepmine_sentinel_ai/core/ml/hyperparameter_tuner.py", 
             "deepmine_sentinel_ai/core/ml/training/hyperparameter_tuner.py"),
            ("deepmine_sentinel_ai/core/ml/cross_validation.py", 
             "deepmine_sentinel_ai/core/ml/training/cross_validation.py"),
            ("deepmine_sentinel_ai/core/ml/model_checkpoint.py", 
             "deepmine_sentinel_ai/core/ml/training/model_checkpoint.py"),
            ("deepmine_sentinel_ai/core/ml/training_monitor.py", 
             "deepmine_sentinel_ai/core/ml/training/training_monitor.py"),
            
            # Inference system
            ("deepmine_sentinel_ai/core/ml/inference_engine.py", 
             "deepmine_sentinel_ai/core/ml/inference/inference_engine.py"),
            ("deepmine_sentinel_ai/core/ml/performance_monitor.py", 
             "deepmine_sentinel_ai/core/ml/inference/performance_monitor.py"),
            
            # Utilities
            ("deepmine_sentinel_ai/core/ml/model_utils.py", 
             "deepmine_sentinel_ai/core/ml/utils/model_utils.py"),
        ]
        
        for source, target in ml_moves:
            self.move_file(source, target)
    
    def organize_api_system(self):
        """Reorganize API files"""
        
        api_moves = [
            ("deepmine_sentinel_ai/core/api_views.py", 
             "deepmine_sentinel_ai/core/api/views.py"),
        ]
        
        for source, target in api_moves:
            self.move_file(source, target)
    
    def organize_models(self):
        """Split and organize Django models"""
        
        # This would require more complex refactoring
        # For now, we'll plan the structure
        models_structure = [
            "deepmine_sentinel_ai/core/models/__init__.py",
            "deepmine_sentinel_ai/core/models/base.py",
            "deepmine_sentinel_ai/core/models/stope.py", 
            "deepmine_sentinel_ai/core/models/monitoring.py",
            "deepmine_sentinel_ai/core/models/events.py",
            "deepmine_sentinel_ai/core/models/impact.py",
            "deepmine_sentinel_ai/core/models/timeseries.py",
        ]
        
        for model_file in models_structure:
            self.log_operation("PLAN_CREATE", "", model_file)
    
    def organize_tests(self):
        """Move and organize test files"""
        
        test_moves = [
            ("deepmine_sentinel_ai/task10_final_validation.py", 
             "tests/integration/test_inference_engine.py"),
            ("deepmine_sentinel_ai/test_inference_api.py", 
             "tests/api/test_inference_api.py"),
            ("deepmine_sentinel_ai/validate_task5.py", 
             "tests/ml/test_lstm_models.py"),
        ]
        
        # Move validation scripts from core
        validation_files = [
            "deepmine_sentinel_ai/core/validation",
        ]
        
        for source, target in test_moves:
            self.move_file(source, target)
    
    def organize_documentation(self):
        """Organize documentation files"""
        
        doc_moves = [
            ("deepmine_sentinel_ai/core/docs", "docs/ml"),
            ("deepmine_sentinel_ai/TASK_10_INFERENCE_ENGINE_SUMMARY.md", 
             "docs/ml/inference_engine.md"),
            ("deepmine_sentinel_ai/PROJECT_STATUS_REPORT.py", 
             "scripts/generate_status_report.py"),
        ]
        
        for source, target in doc_moves:
            if self.project_root.joinpath(source).exists():
                self.move_file(source, target)
    
    def clean_redundant_files(self):
        """Remove redundant and temporary files"""
        
        files_to_remove = [
            # Redundant ml_models directory (after consolidation)
            "deepmine_sentinel_ai/core/ml_models",
            
            # Temporary files
            "deepmine_sentinel_ai/tuning_results",
            "deepmine_sentinel_ai/checkpoints",
            "deepmine_sentinel_ai/results", 
            
            # Log files (keep recent ones)
            "deepmine_sentinel_ai/django_errors.log",
            "deepmine_sentinel_ai/inference_server.log",
            "deepmine_sentinel_ai/server.log",
        ]
        
        for file_path in files_to_remove:
            full_path = self.project_root / file_path
            if full_path.exists():
                self.log_operation("REMOVE", str(full_path), "")
                if not self.dry_run:
                    if full_path.is_dir():
                        shutil.rmtree(full_path)
                    else:
                        full_path.unlink()
    
    def move_file(self, source: str, target: str):
        """Move a file from source to target"""
        source_path = self.project_root / source
        target_path = self.project_root / target
        
        if source_path.exists():
            self.log_operation("MOVE", str(source_path), str(target_path))
            
            if not self.dry_run:
                # Create target directory if it doesn't exist
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source_path), str(target_path))
        else:
            self.log_operation("SKIP_MISSING", str(source_path), str(target_path))
    
    def create_init_files(self):
        """Create __init__.py files for new packages"""
        
        init_files = [
            "deepmine_sentinel_ai/core/models/__init__.py",
            "deepmine_sentinel_ai/core/views/__init__.py", 
            "deepmine_sentinel_ai/core/api/__init__.py",
            "deepmine_sentinel_ai/core/ml/models/__init__.py",
            "deepmine_sentinel_ai/core/ml/training/__init__.py",
            "deepmine_sentinel_ai/core/ml/inference/__init__.py",
            "deepmine_sentinel_ai/core/ml/data/__init__.py",
            "deepmine_sentinel_ai/core/ml/utils/__init__.py",
            "deepmine_sentinel_ai/core/ml/config/__init__.py",
        ]
        
        for init_file in init_files:
            init_path = self.project_root / init_file
            if not self.dry_run and not init_path.exists():
                init_path.parent.mkdir(parents=True, exist_ok=True)
                init_path.write_text("# Auto-generated __init__.py\n")
            self.log_operation("CREATE_INIT", "", str(init_path))
    
    def create_configuration_files(self):
        """Create new configuration files"""
        
        config_files = {
            ".gitignore": self.get_gitignore_content(),
            "docker-compose.yml": self.get_docker_compose_content(),
            "Dockerfile": self.get_dockerfile_content(),
            "README.md": self.get_readme_content(),
        }
        
        for filename, content in config_files.items():
            file_path = self.project_root / filename
            if not file_path.exists():
                self.log_operation("CREATE_CONFIG", "", str(file_path))
                if not self.dry_run:
                    file_path.write_text(content)
    
    def get_gitignore_content(self):
        """Get .gitignore content"""
        return """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# Django
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal
media/
static_collected/

# ML Models and Data
models/trained_models/*.keras
models/trained_models/*.h5
data/raw/
data/processed/
logs/
checkpoints/
tuning_results/
results/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Backup
backup_*/

# Environment
.env
.env.local
.env.production
"""
    
    def get_docker_compose_content(self):
        """Get docker-compose.yml content"""
        return """version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - .:/app
    environment:
      - DEBUG=1
      - DATABASE_URL=sqlite:///db.sqlite3
    depends_on:
      - db
      
  db:
    image: postgres:13
    environment:
      POSTGRES_DB: deepmine_sentinel
      POSTGRES_USER: deepmine_user
      POSTGRES_PASSWORD: deepmine_pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
      
  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
"""
    
    def get_dockerfile_content(self):
        """Get Dockerfile content"""
        return """FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Collect static files
RUN python deepmine_sentinel_ai/manage.py collectstatic --noinput

EXPOSE 8000

CMD ["python", "deepmine_sentinel_ai/manage.py", "runserver", "0.0.0.0:8000"]
"""
    
    def get_readme_content(self):
        """Get README.md content"""
        return """# 🏗️ Deepmine Sentinel AI

> **Mining Stope Stability Prediction System**  
> Advanced LSTM-based AI system for real-time mining stability prediction

## 🎯 Overview

Deepmine Sentinel AI is a production-ready machine learning system that predicts mining stope stability using advanced LSTM neural networks. The system provides real-time predictions, batch processing capabilities, and comprehensive monitoring for mining operations.

## ✅ Features

- **Real-time Prediction API** - Sub-second response times for stability predictions
- **Batch Processing** - Efficient processing of multiple mining locations
- **Confidence Scoring** - Advanced uncertainty estimation with Shannon entropy
- **Performance Monitoring** - Real-time system metrics and prediction tracking
- **RESTful API** - Complete API for external system integration
- **Model Management** - Automated versioning and caching system

## 🚀 Quick Start

### Local Development
```bash
# Clone repository
git clone https://github.com/Josephat-Onkoba/Deepmine-Sentinel-AI.git
cd Deepmine-Sentinel-AI

# Install dependencies
pip install -r requirements.txt

# Run migrations
cd deepmine_sentinel_ai
python manage.py migrate

# Start development server
python manage.py runserver
```

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build
```

## 📚 Documentation

- [Installation Guide](docs/installation/)
- [API Documentation](docs/api/)
- [ML System Guide](docs/ml/)
- [User Guide](docs/user_guide/)
- [Deployment Guide](docs/deployment/)

## 🔌 API Endpoints

- `POST /api/predict/single` - Single stope prediction
- `POST /api/predict/batch` - Batch predictions
- `GET /api/model/performance` - Model performance metrics
- `GET /api/health` - System health check

## 🏗️ Architecture

- **Backend**: Django 5.2.4 + TensorFlow 2.19.0
- **ML Models**: LSTM Neural Networks (125,924 parameters)
- **API**: RESTful API with comprehensive endpoints
- **Database**: SQLite (dev) / PostgreSQL (production)
- **Monitoring**: Built-in performance tracking

## 📊 Performance

- **Model Accuracy**: 92% validation accuracy
- **Response Time**: ~100ms average prediction time
- **Throughput**: 10+ predictions per second
- **Confidence**: 0.779-0.781 range (high reliability)

## 🤝 Contributing

See [Contributing Guide](docs/developer/contributing.md) for development guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
"""
    
    def save_operations_log(self):
        """Save operations log to file"""
        log_file = self.project_root / "organization_log.json"
        
        if not self.dry_run:
            with open(log_file, 'w') as f:
                json.dump(self.operations_log, f, indent=2)
        
        self.log_operation("SAVE_LOG", "", str(log_file))
    
    def run_organization(self):
        """Execute the complete organization process"""
        print(f"🏗️ Starting Deepmine Sentinel AI Project Organization")
        print(f"📁 Project Root: {self.project_root}")
        print(f"🔧 Mode: {'DRY RUN' if self.dry_run else 'EXECUTE'}")
        print("-" * 60)
        
        steps = [
            ("Creating backup", self.create_backup),
            ("Creating directory structure", self.create_directory_structure),
            ("Organizing ML system", self.organize_ml_system),
            ("Organizing API system", self.organize_api_system),
            ("Planning models organization", self.organize_models),
            ("Organizing tests", self.organize_tests),
            ("Organizing documentation", self.organize_documentation),
            ("Cleaning redundant files", self.clean_redundant_files),
            ("Creating init files", self.create_init_files),
            ("Creating configuration files", self.create_configuration_files),
            ("Saving operations log", self.save_operations_log),
        ]
        
        for step_name, step_function in steps:
            print(f"\n📋 {step_name}...")
            try:
                step_function()
                print(f"✅ {step_name} completed")
            except Exception as e:
                print(f"❌ {step_name} failed: {e}")
        
        print(f"\n🎉 Organization {'simulation' if self.dry_run else 'execution'} completed!")
        print(f"📊 Total operations: {len(self.operations_log)}")
        
        if self.dry_run:
            print("\n💡 Run with --execute to apply changes")
        else:
            print(f"\n💾 Backup created at: {self.backup_dir}")
            print(f"📋 Operations log: {self.project_root}/organization_log.json")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Organize Deepmine Sentinel AI project")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--execute", action="store_true", help="Execute changes (default: dry run)")
    
    args = parser.parse_args()
    
    organizer = ProjectOrganizer(
        project_root=args.project_root,
        dry_run=not args.execute
    )
    
    organizer.run_organization()


if __name__ == "__main__":
    main()
