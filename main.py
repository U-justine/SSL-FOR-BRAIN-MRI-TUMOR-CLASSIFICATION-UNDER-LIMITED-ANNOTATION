"""
Main Script - Brain MRI Tumor Classification
Complete pipeline for supervised and SSL experiments
"""

import torch
import os
from config import Config
from dataset import get_dataloaders
from model import SupervisedClassifier, SimCLRv1, SimCLRv2, FineTunedClassifier
from train import train_supervised, train_ssl, train_finetuned
from evaluate import evaluate_model, print_results, save_confusion_matrix, save_results_table, create_comparison_plot


def main():
    """Main execution pipeline."""
    
    print("\n" + "="*80)
    print("BRAIN MRI TUMOR CLASSIFICATION WITH SELF-SUPERVISED LEARNING")
    print("GPU-Optimized Implementation")
    print("="*80)
    
    # Initialize configuration
    config = Config()
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: GPU not available. Training will be slow on CPU.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Get data loaders
    print("\n" + "-"*80)
    print("STEP 1: Loading Dataset")
    print("-"*80)
    
    _, test_loader, ssl_loader = get_dataloaders(config)
    
    # Print dataset info
    print(f"✓ Dataset loaded successfully")
    print(f"  SSL batch size: {config.ssl_batch_size}")
    print(f"  Training batch size: {config.supervised_batch_size}")
    
    # =========================================================================
    # STEP 2: SSL PRETRAINING
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 2: SSL Pretraining")
    print("-"*80)
    
    # Train SimCLR v1
    print("\n[2.1] Pretraining SimCLR v1...")
    simclr_v1 = SimCLRv1(config)
    simclr_v1 = train_ssl(simclr_v1, ssl_loader, config, "simclr_v1", config.ssl_epochs_v1)
    
    # Train SimCLR v2
    print("\n[2.2] Pretraining SimCLR v2...")
    simclr_v2 = SimCLRv2(config)
    simclr_v2 = train_ssl(simclr_v2, ssl_loader, config, "simclr_v2", config.ssl_epochs_v2)
    
    # =========================================================================
    # STEP 3: EXPERIMENTS ACROSS LABEL PERCENTAGES
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 3: Running Experiments Across Label Percentages")
    print("-"*80)
    
    all_results = []
    
    for label_pct in config.label_percentages:
        print(f"\n{'='*80}")
        print(f"EXPERIMENTS AT {label_pct*100}% LABELS")
        print(f"{'='*80}")
        
        # Get data loaders for this label percentage
        train_loader, test_loader_eval, _ = get_dataloaders(config, label_pct)
        
        # ─────────────────────────────────────────────────────────────────────
        # Experiment 1: Supervised from scratch
        # ─────────────────────────────────────────────────────────────────────
        print(f"\n[1/3] Training supervised baseline...")
        supervised = SupervisedClassifier(config)
        supervised = train_supervised(supervised, train_loader, test_loader_eval, config, 
                                     "supervised", label_pct)
        
        # Evaluate
        results = evaluate_model(supervised, test_loader_eval, config)
        results['model'] = 'supervised'
        results['label_pct'] = f"{label_pct*100}%"
        all_results.append(results)
        print_results(results, config.class_names)
        
        # Save confusion matrix
        cm_path = os.path.join(config.output_dir, f"cm_supervised_{int(label_pct*100)}pct.png")
        save_confusion_matrix(results['confusion_matrix'], config.class_names, cm_path)
        
        # Clear GPU memory
        del supervised
        torch.cuda.empty_cache()
        
        # ─────────────────────────────────────────────────────────────────────
        # Experiment 2: SimCLR v1 + Fine-tuning
        # ─────────────────────────────────────────────────────────────────────
        print(f"\n[2/3] Fine-tuning SimCLR v1...")
        simclr_v1_ft = FineTunedClassifier(simclr_v1, config.num_classes, config.freeze_backbone)
        simclr_v1_ft = train_finetuned(simclr_v1_ft, train_loader, test_loader_eval, config,
                                       "simclr_v1_ft", label_pct)
        
        # Evaluate
        results = evaluate_model(simclr_v1_ft, test_loader_eval, config)
        results['model'] = 'simclr_v1_ft'
        results['label_pct'] = f"{label_pct*100}%"
        all_results.append(results)
        print_results(results, config.class_names)
        
        # Save confusion matrix
        cm_path = os.path.join(config.output_dir, f"cm_simclr_v1_{int(label_pct*100)}pct.png")
        save_confusion_matrix(results['confusion_matrix'], config.class_names, cm_path)
        
        # Clear GPU memory
        del simclr_v1_ft
        torch.cuda.empty_cache()
        
        # ─────────────────────────────────────────────────────────────────────
        # Experiment 3: SimCLR v2 + Fine-tuning
        # ─────────────────────────────────────────────────────────────────────
        print(f"\n[3/3] Fine-tuning SimCLR v2...")
        simclr_v2_ft = FineTunedClassifier(simclr_v2, config.num_classes, config.freeze_backbone)
        simclr_v2_ft = train_finetuned(simclr_v2_ft, train_loader, test_loader_eval, config,
                                       "simclr_v2_ft", label_pct)
        
        # Evaluate
        results = evaluate_model(simclr_v2_ft, test_loader_eval, config)
        results['model'] = 'simclr_v2_ft'
        results['label_pct'] = f"{label_pct*100}%"
        all_results.append(results)
        print_results(results, config.class_names)
        
        # Save confusion matrix
        cm_path = os.path.join(config.output_dir, f"cm_simclr_v2_{int(label_pct*100)}pct.png")
        save_confusion_matrix(results['confusion_matrix'], config.class_names, cm_path)
        
        # Clear GPU memory
        del simclr_v2_ft
        torch.cuda.empty_cache()
    
    # =========================================================================
    # STEP 4: SAVE RESULTS
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 4: Saving Results")
    print("-"*80)
    
    # Save results table
    save_results_table(all_results, config)
    
    # Create comparison plot
    create_comparison_plot(all_results, config)
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETE!")
    print("="*80)
    
    print(f"\nResults saved in: {config.output_dir}/")
    print(f"  - Confusion matrices: cm_*.png")
    print(f"  - Results table: results_summary.txt")
    print(f"  - Comparison plot: comparison_plot.png")
    print(f"\nCheckpoints saved in: {config.checkpoint_dir}/")
    
    # Print best results
    print("\n" + "-"*80)
    print("BEST RESULTS")
    print("-"*80)
    
    for model_type in ['supervised', 'simclr_v1_ft', 'simclr_v2_ft']:
        model_results = [r for r in all_results if r['model'] == model_type]
        if model_results:
            best = max(model_results, key=lambda x: x['accuracy'])
            print(f"\n{model_type}:")
            print(f"  Best accuracy: {best['accuracy']:.2f}% at {best['label_pct']} labels")
    
    print("\n" + "="*80)
    print("✓ ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
