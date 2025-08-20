#!/usr/bin/env python3
"""
Demo: ML Evaluation and Advanced Analytics for Experiment Set E
Demonstrates machine learning, hazard curves, and comprehensive evaluation
"""

import sys
import traceback
from enhanced_temporal_query_engine import EnhancedTemporalQueryEngine

def demo_ml_training():
    """Demonstrate ML training with isotonic calibration"""
    print("\n🤖 MACHINE LEARNING TRAINING")
    print("=" * 60)
    print("📋 One-vs-rest classifiers with isotonic calibration")
    print("🎯 Features: f8_q, f8_slope_sign, HTF f47-f50, archaeological significance")
    print("=" * 60)
    
    try:
        engine = EnhancedTemporalQueryEngine()
        
        # Test ML training queries
        test_queries = [
            "Train machine learning models for path prediction",
            "Show me ML predictions with isotonic calibration",
            "Analyze cross-validation scores for path classifiers"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}️⃣ Query: '{query}'")
            print("-" * 50)
            
            result = engine.ask(query)
            print_ml_results(result)
            
    except Exception as e:
        print(f"❌ ML Training Error: {e}")
        traceback.print_exc()

def demo_hazard_curves():
    """Demonstrate hazard curve and survival analysis"""
    print("\n📈 HAZARD CURVE ANALYSIS")
    print("=" * 60)
    print("📋 Time-to-event modeling with survival analysis")
    print("🎯 Resolution timing: CONT→80%, MR→mid, ACCEL→80%")
    print("=" * 60)
    
    try:
        engine = EnhancedTemporalQueryEngine()
        
        # Test hazard curve queries
        test_queries = [
            "Analyze hazard curves for path resolution timing",
            "Show me survival analysis for different paths",
            "What are the median resolution times for each path?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}️⃣ Query: '{query}'")
            print("-" * 50)
            
            result = engine.ask(query)
            print_hazard_results(result)
            
    except Exception as e:
        print(f"❌ Hazard Analysis Error: {e}")
        traceback.print_exc()

def demo_model_evaluation():
    """Demonstrate comprehensive model evaluation"""
    print("\n📊 MODEL EVALUATION")
    print("=" * 60)
    print("📋 Confusion matrix, precision, recall, F1-score")
    print("🎯 Per-path performance metrics with statistical validation")
    print("=" * 60)
    
    try:
        engine = EnhancedTemporalQueryEngine()
        
        # Test evaluation queries
        test_queries = [
            "Evaluate model performance with confusion matrix",
            "Show me precision and recall for each path type",
            "Generate comprehensive evaluation metrics"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}️⃣ Query: '{query}'")
            print("-" * 50)
            
            result = engine.ask(query)
            print_evaluation_results(result)
            
    except Exception as e:
        print(f"❌ Model Evaluation Error: {e}")
        traceback.print_exc()

def demo_feature_attribution():
    """Demonstrate feature importance and attribution analysis"""
    print("\n🔍 FEATURE ATTRIBUTION ANALYSIS")
    print("=" * 60)
    print("📋 Feature importance for path selection decisions")
    print("🎯 Drivers vs inhibitors for CONT/MR/ACCEL paths")
    print("=" * 60)
    
    try:
        engine = EnhancedTemporalQueryEngine()
        
        # Test feature attribution queries
        test_queries = [
            "Analyze feature importance for path selection",
            "Show me feature attributions for CONT vs MR decisions",
            "What features drive ACCEL path classification?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}️⃣ Query: '{query}'")
            print("-" * 50)
            
            result = engine.ask(query)
            print_feature_attribution_results(result)
            
    except Exception as e:
        print(f"❌ Feature Attribution Error: {e}")
        traceback.print_exc()

def demo_comprehensive_analytics():
    """Demonstrate comprehensive analytics combining all components"""
    print("\n🎯 COMPREHENSIVE ANALYTICS PIPELINE")
    print("=" * 80)
    print("📋 End-to-end ML pipeline: Training → Evaluation → Feature Analysis → Hazard Curves")
    print("=" * 80)
    
    try:
        engine = EnhancedTemporalQueryEngine()
        
        print("🔄 Running comprehensive analytics pipeline...")
        
        # Step 1: Train ML models
        print("\n1️⃣ Training ML Models...")
        ml_result = engine.ask("Train machine learning models for path prediction")
        
        # Step 2: Evaluate performance
        print("\n2️⃣ Evaluating Model Performance...")
        eval_result = engine.ask("Evaluate model performance with confusion matrix")
        
        # Step 3: Analyze feature importance
        print("\n3️⃣ Analyzing Feature Importance...")
        feat_result = engine.ask("Analyze feature importance for path selection")
        
        # Step 4: Generate hazard curves
        print("\n4️⃣ Generating Hazard Curves...")
        hazard_result = engine.ask("Analyze hazard curves for path resolution timing")
        
        # Comprehensive summary
        print(f"\n📈 COMPREHENSIVE ANALYTICS SUMMARY")
        print("=" * 60)
        
        # Extract key metrics
        ml_samples = ml_result.get("training_results", {}).get("total_samples", 0)
        ml_classes = ml_result.get("training_results", {}).get("class_distribution", {})
        
        eval_accuracy = eval_result.get("evaluation_results", {}).get("overall_accuracy", 0)
        eval_metrics = eval_result.get("evaluation_results", {}).get("path_metrics", {})
        
        hazard_stats = hazard_result.get("hazard_results", {}).get("path_hazard_analysis", {})
        
        print(f"🤖 ML Training:")
        print(f"   Samples: {ml_samples}")
        print(f"   Classes: {ml_classes}")
        
        print(f"📊 Model Performance:")
        print(f"   Overall Accuracy: {eval_accuracy:.1%}")
        for path, metrics in eval_metrics.items():
            f1 = metrics.get("f1_score", 0)
            support = metrics.get("support", 0)
            print(f"   {path}: F1={f1:.2f} (n={support})")
        
        print(f"⏱️ Hazard Analysis:")
        for path, stats in hazard_stats.items():
            resolution_rate = stats.get("resolution_rate", 0)
            median_time = stats.get("median_time", 0)
            print(f"   {path}: {resolution_rate:.1%} resolution, {median_time:.1f}min median")
        
        # Feature insights
        if feat_result.get("attributions"):
            print(f"🔍 Key Features:")
            for path, attribution in feat_result["attributions"].items():
                top_features = attribution.get("top_positive_features", [])[:2]
                if top_features:
                    print(f"   {path} drivers: {', '.join(top_features)}")
        
        print(f"\n✅ Comprehensive analytics pipeline complete!")
        
    except Exception as e:
        print(f"❌ Comprehensive Analytics Error: {e}")
        traceback.print_exc()

def print_ml_results(result: dict):
    """Print ML training results"""
    if not result:
        print("  ❌ No ML results returned")
        return
    
    print(f"  📋 Analysis Type: {result.get('query_type', 'unknown')}")
    
    # Training results
    training_results = result.get("training_results", {})
    if training_results:
        total_samples = training_results.get("total_samples", 0)
        class_dist = training_results.get("class_distribution", {})
        cv_scores = training_results.get("cross_validation_scores", {})
        
        print(f"  🤖 Training Results:")
        print(f"     Total Samples: {total_samples}")
        print(f"     Class Distribution: {class_dist}")
        
        if cv_scores:
            print(f"  📊 Cross-Validation AUC Scores:")
            for path, scores in cv_scores.items():
                mean_auc = scores.get("mean_auc", 0)
                std_auc = scores.get("std_auc", 0)
                print(f"     {path}: {mean_auc:.3f} ± {std_auc:.3f}")
    
    # Sample predictions
    predictions = result.get("predictions", {}).get("sample_predictions", {})
    if predictions:
        print(f"  🎯 Sample Predictions:")
        for path, probs in predictions.items():
            if probs:
                avg_prob = sum(probs) / len(probs)
                print(f"     {path}: {avg_prob:.3f} avg probability")
    
    # Insights
    insights = result.get("insights", [])
    if insights:
        print(f"  💡 Key Insights:")
        for insight in insights[:3]:
            print(f"     • {insight}")
    
    print()

def print_hazard_results(result: dict):
    """Print hazard curve analysis results"""
    if not result:
        print("  ❌ No hazard results returned")
        return
    
    print(f"  📋 Analysis Type: {result.get('query_type', 'unknown')}")
    
    # Hazard analysis
    hazard_results = result.get("hazard_results", {})
    path_hazard = hazard_results.get("path_hazard_analysis", {})
    
    if path_hazard:
        print(f"  📈 Hazard Analysis:")
        for path, stats in path_hazard.items():
            total_events = stats.get("total_events", 0)
            resolved = stats.get("resolved_events", 0)
            resolution_rate = stats.get("resolution_rate", 0)
            median_time = stats.get("median_time", 0)
            
            print(f"     {path}: {resolved}/{total_events} resolved ({resolution_rate:.1%}), {median_time:.1f}min median")
    
    # Median resolution times
    median_times = hazard_results.get("median_resolution_times", {})
    if median_times:
        print(f"  ⏱️ Median Resolution Times:")
        for path, time in median_times.items():
            print(f"     {path}: {time:.1f} minutes")
    
    # Survival probabilities
    survival = hazard_results.get("comparative_survival", {})
    if survival:
        print(f"  📊 Survival Probabilities:")
        for path, probs in survival.items():
            prob_60 = probs.get("t_60min", "N/A")
            prob_120 = probs.get("t_120min", "N/A")
            print(f"     {path}: 60min={prob_60}, 120min={prob_120}")
    
    # Insights
    insights = result.get("insights", [])
    if insights:
        print(f"  💡 Key Insights:")
        for insight in insights[:3]:
            print(f"     • {insight}")
    
    print()

def print_evaluation_results(result: dict):
    """Print model evaluation results"""
    if not result:
        print("  ❌ No evaluation results returned")
        return
    
    print(f"  📋 Analysis Type: {result.get('query_type', 'unknown')}")
    
    # Evaluation results
    eval_results = result.get("evaluation_results", {})
    
    # Overall accuracy
    overall_accuracy = eval_results.get("overall_accuracy", 0)
    print(f"  🎯 Overall Accuracy: {overall_accuracy:.1%}")
    
    # Path metrics
    path_metrics = eval_results.get("path_metrics", {})
    if path_metrics:
        print(f"  📊 Per-Path Metrics:")
        for path, metrics in path_metrics.items():
            precision = metrics.get("precision", 0)
            recall = metrics.get("recall", 0)
            f1_score = metrics.get("f1_score", 0)
            support = metrics.get("support", 0)
            
            print(f"     {path}: P={precision:.2f}, R={recall:.2f}, F1={f1_score:.2f} (n={support})")
    
    # Confusion matrix
    confusion_matrix = eval_results.get("confusion_matrix", [])
    if confusion_matrix:
        print(f"  📈 Confusion Matrix:")
        labels = ["CONT", "MR", "ACCEL"]
        for i, row in enumerate(confusion_matrix):
            if i < len(labels):
                print(f"     {labels[i]}: {row}")
    
    # Insights
    insights = result.get("insights", [])
    if insights:
        print(f"  💡 Key Insights:")
        for insight in insights[:3]:
            print(f"     • {insight}")
    
    print()

def print_feature_attribution_results(result: dict):
    """Print feature attribution results"""
    if not result:
        print("  ❌ No feature attribution results returned")
        return
    
    print(f"  📋 Analysis Type: {result.get('query_type', 'unknown')}")
    
    # Attribution results
    attributions = result.get("attributions", {})
    
    if attributions:
        print(f"  🔍 Feature Attributions:")
        for path, attribution in attributions.items():
            top_positive = attribution.get("top_positive_features", [])[:3]
            top_negative = attribution.get("top_negative_features", [])[:3]
            
            if top_positive or top_negative:
                print(f"     {path}:")
                if top_positive:
                    print(f"       Drivers: {', '.join(top_positive)}")
                if top_negative:
                    print(f"       Inhibitors: {', '.join(top_negative)}")
    
    # Insights
    insights = result.get("insights", [])
    if insights:
        print(f"  💡 Key Insights:")
        for insight in insights[:3]:
            print(f"     • {insight}")
    
    print()

def main():
    """Main demonstration function"""
    print("🚀 IRONFORGE: ML Evaluation & Advanced Analytics Demonstration")
    print("🎯 Machine Learning, Hazard Curves, and Statistical Validation")
    print("=" * 80)
    
    try:
        # Run all demonstration modules
        demo_ml_training()
        demo_hazard_curves()
        demo_model_evaluation()
        demo_feature_attribution()
        demo_comprehensive_analytics()
        
        print("\n✅ ML Evaluation & Advanced Analytics Demonstration Complete!")
        print("🎯 The Enhanced Temporal Query Engine now provides:")
        print("   • Machine learning with isotonic calibration")
        print("   • Hazard curve analysis and survival modeling")
        print("   • Comprehensive model evaluation (confusion matrix, metrics)")
        print("   • Feature attribution and importance analysis")
        print("   • End-to-end analytics pipeline integration")
        print("   • Statistical rigor with cross-validation")
        print("   • Time-to-event modeling for path resolution")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure all dependencies are available:")
        print("   - enhanced_temporal_query_engine.py")
        print("   - experiment_e_analyzer.py")
        print("   - ml_path_predictor.py")
        print("   - sklearn (for ML components)")
        print("   - scipy (for statistical analysis)")
        print("   - lifelines (optional, for survival analysis)")
        
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()