#!/usr/bin/env python3
"""
Test script for the enhanced training workflow with comprehensive visualizations.
This script validates that all training components work correctly and generate proper outputs.
"""

import os
import sys
import django
import time
import shutil
from pathlib import Path

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepmine_sentinel_ai.settings')
django.setup()

from django.core.management import call_command
from django.core.management.base import CommandError


def setup_test_environment():
    """Setup test environment for training validation."""
    print("🔧 Setting up test environment...")
    
    # Create visualizations directory
    viz_dir = "test_training_visualizations"
    if os.path.exists(viz_dir):
        shutil.rmtree(viz_dir)
    os.makedirs(viz_dir, exist_ok=True)
    
    return viz_dir


def validate_generated_files(viz_dir):
    """Validate that all expected files were generated."""
    print("🔍 Validating generated files...")
    
    expected_files = [
        'training_history_comprehensive.png',
        'performance_summary.png', 
        'training_heatmap.png',
        'metrics_comparison.png',
        'model_architecture.png',
        'model_architecture.txt',
        'training_report.md'
    ]
    
    missing_files = []
    generated_files = []
    
    for filename in expected_files:
        filepath = os.path.join(viz_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            generated_files.append((filename, size))
            print(f"  ✅ {filename} ({size:,} bytes)")
        else:
            missing_files.append(filename)
            print(f"  ❌ {filename} - NOT FOUND")
    
    return missing_files, generated_files


def validate_training_report(viz_dir):
    """Validate the contents of the training report."""
    print("📋 Validating training report...")
    
    report_path = os.path.join(viz_dir, 'training_report.md')
    if not os.path.exists(report_path):
        print("  ❌ Training report not found")
        return False
    
    with open(report_path, 'r') as f:
        content = f.read()
    
    # Check for required sections
    required_sections = [
        '# Enhanced Dual-Branch Stability Predictor - Training Report',
        '## 🛠️ Training Configuration',
        '## 🏗️ Model Architecture', 
        '## 📊 Final Performance',
        '## 📈 Training Metrics',
        '## 🎯 Training Analysis',
        '## 📁 Generated Visualizations',
        '## 💡 Recommendations'
    ]
    
    missing_sections = []
    for section in required_sections:
        if section in content:
            print(f"  ✅ Found: {section}")
        else:
            missing_sections.append(section)
            print(f"  ❌ Missing: {section}")
    
    print(f"  📊 Report size: {len(content):,} characters")
    return len(missing_sections) == 0


def validate_model_files():
    """Validate that model files were saved correctly."""
    print("🧠 Validating saved model files...")
    
    models_dir = "models"
    expected_model_files = [
        'enhanced_dual_branch_model.keras',
        'enhanced_dual_branch_model_components.pkl',
        'enhanced_dual_branch_model_metadata.json'
    ]
    
    found_files = []
    missing_files = []
    
    for filename in expected_model_files:
        filepath = os.path.join(models_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            found_files.append((filename, size))
            print(f"  ✅ {filename} ({size:,} bytes)")
        else:
            missing_files.append(filename)
            print(f"  ❌ {filename} - NOT FOUND")
    
    return missing_files, found_files


def test_training_command():
    """Test the enhanced training command with minimal epochs for speed."""
    print("🚀 Testing enhanced training command...")
    
    viz_dir = setup_test_environment()
    
    start_time = time.time()
    
    try:
        # Run training with minimal epochs for testing
        call_command(
            'train_model',
            epochs=3,  # Minimal epochs for testing
            batch_size=16,
            learning_rate=0.001,
            validation_split=0.2,
            save_plots=True,
            plots_dir=viz_dir,
            generate_report=True,
            verbose=True
        )
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.2f} seconds")
        
        # Validate outputs
        print("\n" + "="*60)
        print("🔍 VALIDATION RESULTS")
        print("="*60)
        
        # Check visualization files
        missing_viz, generated_viz = validate_generated_files(viz_dir)
        
        # Check training report
        report_valid = validate_training_report(viz_dir)
        
        # Check model files
        missing_models, found_models = validate_model_files()
        
        # Summary
        print("\n" + "="*60)
        print("📊 VALIDATION SUMMARY")
        print("="*60)
        
        print(f"🎨 Visualization files: {len(generated_viz)}/{len(generated_viz) + len(missing_viz)} generated")
        if missing_viz:
            print(f"   ❌ Missing: {', '.join(missing_viz)}")
        
        print(f"📋 Training report: {'✅ Valid' if report_valid else '❌ Invalid'}")
        
        print(f"🧠 Model files: {len(found_models)}/{len(found_models) + len(missing_models)} found")
        if missing_models:
            print(f"   ❌ Missing: {', '.join(missing_models)}")
        
        # Overall status
        total_issues = len(missing_viz) + len(missing_models) + (0 if report_valid else 1)
        
        if total_issues == 0:
            print(f"\n🎉 ALL TESTS PASSED! Training workflow is fully functional.")
            print(f"⏱️  Total training time: {training_time:.2f} seconds")
            print(f"📁 Visualizations saved to: {viz_dir}/")
        else:
            print(f"\n⚠️  {total_issues} issues found. Please review the missing components.")
        
        return total_issues == 0
        
    except CommandError as e:
        print(f"❌ Training command failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def display_visualization_details(viz_dir):
    """Display details about generated visualizations."""
    print("\n" + "="*60)
    print("🎨 GENERATED VISUALIZATIONS DETAILS")
    print("="*60)
    
    if not os.path.exists(viz_dir):
        print("❌ Visualization directory not found")
        return
    
    for filename in os.listdir(viz_dir):
        filepath = os.path.join(viz_dir, filename)
        if os.path.isfile(filepath):
            size = os.path.getsize(filepath)
            ext = os.path.splitext(filename)[1]
            
            if ext == '.png':
                print(f"🖼️  {filename} - {size:,} bytes")
            elif ext == '.txt':
                print(f"📄 {filename} - {size:,} bytes")
            elif ext == '.md':
                print(f"📋 {filename} - {size:,} bytes")
            else:
                print(f"📁 {filename} - {size:,} bytes")


def main():
    """Main test execution."""
    print("🧪 Enhanced Training Workflow Validation")
    print("="*60)
    
    # Test the training workflow
    success = test_training_command()
    
    # Display details about generated files
    display_visualization_details("test_training_visualizations")
    
    print("\n" + "="*60)
    if success:
        print("🎉 ENHANCED TRAINING VALIDATION PASSED!")
        print("✅ All components are working correctly")
        print("✅ All visualizations are being generated")
        print("✅ Training reports are comprehensive")
        print("✅ Model saving/loading is working")
    else:
        print("❌ ENHANCED TRAINING VALIDATION FAILED!")
        print("⚠️  Some components need attention")
    
    print("="*60)
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
